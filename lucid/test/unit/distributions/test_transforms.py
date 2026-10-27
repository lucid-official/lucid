"""The distribution transforms, checked on the two things they promise.

``distributions/transforms.py`` sat at 71.8%.  A transform is worth
having only if ``inv(fwd(x)) == x`` and ``log_abs_det_jacobian`` really
is the log-determinant of its Jacobian — the second is what
``TransformedDistribution.log_prob`` multiplies in, so a wrong one is a
wrong density and everything built on it optimises the wrong objective.

Neither property is checkable by inspection, and both are checkable
exactly: the round-trip against the input, and the log-determinant
against a central-difference Jacobian computed here.  Three of the
transforms failed one of them.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.distributions._kl_cross as kl_cross
import lucid.distributions.transformed_distribution as transformed_distribution
import lucid.distributions.transforms as T
from lucid.distributions._util import _sum_rightmost

X = np.array([0.3, -1.2, 0.7, 2.1])
POS = np.array([0.4, 1.3, 2.7, 0.9])


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float64))


def _v(x):
    return np.asarray(x.numpy())


def _numeric_logdet(fn, x, h=1e-6):
    """``log|det(d fn / d x)|`` by central differences.

    ``fn`` maps a flat vector of free coordinates to a flat vector of the
    same length — the caller picks which output coordinates are free when
    the codomain is a constrained set.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    jac = np.zeros((fn(x).size, n))
    for k in range(n):
        step = np.zeros(n)
        step[k] = h
        jac[:, k] = (fn(x + step) - fn(x - step)) / (2 * h)
    return np.log(np.abs(np.linalg.det(jac)))


# ── the element-wise transforms ───────────────────────────────────────────────

ELEMENTWISE = [
    ("exp", T.ExpTransform(), X),
    ("sigmoid", T.SigmoidTransform(), X),
    ("tanh", T.TanhTransform(), X),
    ("affine", T.AffineTransform(2.0, 3.0), X),
    ("affine with a negative scale", T.AffineTransform(1.0, -2.0), X),
    ("power 2", T.PowerTransform(_t(2.0)), POS),
    ("power 1/2", T.PowerTransform(_t(0.5)), POS),
    (
        "compose exp then affine",
        T.ComposeTransform([T.ExpTransform(), T.AffineTransform(1.0, 2.0)]),
        X,
    ),
    ("compose sigmoid then exp", T.ComposeTransform([T.SigmoidTransform(), T.ExpTransform()]), X),
]


@pytest.mark.parametrize(
    "transform,x", [c[1:] for c in ELEMENTWISE], ids=[c[0] for c in ELEMENTWISE]
)
def test_an_elementwise_transform_round_trips(transform, x):
    assert np.allclose(_v(transform.inv(transform(_t(x)))), x, atol=1e-9)


@pytest.mark.parametrize(
    "transform,x", [c[1:] for c in ELEMENTWISE], ids=[c[0] for c in ELEMENTWISE]
)
def test_an_elementwise_log_det_is_the_log_derivative(transform, x):
    """Element-wise, so the Jacobian is diagonal and the log-determinant
    is the elementwise ``log|dy/dx|`` rather than a sum."""
    h = 1e-6
    numeric = np.log(np.abs((_v(transform(_t(x + h))) - _v(transform(_t(x - h)))) / (2 * h)))
    assert np.allclose(
        _v(transform.log_abs_det_jacobian(_t(x), transform(_t(x)))), numeric, atol=1e-5
    )


def test_the_sign_of_an_affine_scale_does_not_reach_the_log_det():
    """It is ``log|det|``, so a reflection has the same one."""
    up = T.AffineTransform(0.0, 3.0)
    down = T.AffineTransform(0.0, -3.0)
    assert np.allclose(
        _v(up.log_abs_det_jacobian(_t(X), up(_t(X)))),
        _v(down.log_abs_det_jacobian(_t(X), down(_t(X)))),
    )


def test_exp_and_its_inverse_are_the_log():
    assert np.allclose(_v(T.ExpTransform()(_t(X))), np.exp(X))
    assert np.allclose(_v(T.ExpTransform().inv(_t(POS))), np.log(POS))


def test_tanh_maps_into_the_unit_interval_and_saturates_at_its_ends():
    """The codomain is open, but in floating point it is not.

    Past about ``|x| = 19`` in float64 ``tanh`` rounds to exactly ``±1``,
    and the inverse of that is ``±inf`` — so a flow that pushes a sample
    far enough out comes back with an infinite log-density rather than a
    large one.  The reference framework does the same thing, so this is
    the shared contract rather than a Lucid gap; pinned because it is the
    kind of thing found the hard way at 3am.
    """
    transform = T.TanhTransform()
    moderate = _v(transform(_t(np.array([-5.0, 0.0, 5.0]))))
    assert (moderate > -1.0).all() and (moderate < 1.0).all()
    assert np.allclose(_v(transform.inv(_t(moderate))), [-5.0, 0.0, 5.0], atol=1e-9)

    saturated = _v(transform(_t(np.array([-30.0, 30.0]))))
    assert np.array_equal(saturated, np.array([-1.0, 1.0]))
    assert not np.isfinite(_v(transform.inv(_t(saturated)))).any()


# ── the structured transforms ─────────────────────────────────────────────────


def test_stick_breaking_lands_on_the_simplex():
    y = _v(T.StickBreakingTransform()(_t(X[:3])))
    assert y.shape == (4,)
    assert np.isclose(y.sum(), 1.0)
    assert (y > 0).all()


@pytest.mark.parametrize(
    "x", [np.array([0.3, -1.2, 0.7]), np.array([0.0, 0.0]), np.array([1.5, -0.5, 0.2, 0.9])]
)
def test_stick_breaking_round_trips(x):
    transform = T.StickBreakingTransform()
    assert np.allclose(_v(transform.inv(transform(_t(x)))), x, atol=1e-9)


@pytest.mark.parametrize(
    "x",
    [
        np.array([0.3, -1.2, 0.7]),
        np.array([0.0, 0.0]),
        np.array([1.5, -0.5, 0.2, 0.9]),
        np.array([-2.0, 3.0]),
    ],
)
def test_stick_breaking_log_det_is_the_jacobian(x):
    """``y_k = rem_k · z_k``, so ``dy_k/dx_k = rem_k · z_k · (1 - z_k)``
    and the log-determinant is ``Σ log(y_k) + log(1 - z_k)``.

    The ``rem`` factor is already inside ``y_k``.  It was being added a
    second time, which put every simplex density off by ``Σ log(rem_k)``
    — a quantity that varies with the sample, so it does not cancel as a
    normalising constant.
    """
    transform = T.StickBreakingTransform()
    numeric = _numeric_logdet(lambda z: _v(transform(_t(z)))[:-1], x)
    got = float(_v(transform.log_abs_det_jacobian(_t(x), transform(_t(x)))).ravel()[0])
    assert np.isclose(got, numeric, atol=1e-5)


def test_the_stick_breaking_density_is_the_base_density_over_the_jacobian():
    """The identity ``TransformedDistribution`` is built on."""
    transform = T.StickBreakingTransform()
    base = D.Independent(D.Normal(_t(np.zeros(3)), _t(np.ones(3))), 1)
    pushed = T.TransformedDistribution(base, transform)
    for x in np.random.default_rng(0).standard_normal((8, 3)):
        y = transform(_t(x))
        want = float(_v(base.log_prob(_t(x))).ravel()[0]) - float(
            _v(transform.log_abs_det_jacobian(_t(x), y)).ravel()[0]
        )
        assert np.isclose(float(_v(pushed.log_prob(y)).ravel()[0]), want, atol=1e-8)


def test_softmax_lands_on_the_simplex():
    y = _v(T.SoftmaxTransform()(_t(X)))
    assert np.isclose(y.sum(), 1.0)
    assert (y > 0).all()


def test_softmax_inverts_only_up_to_an_additive_constant():
    """``softmax(x) == softmax(x + c)``, so the inverse cannot recover
    ``x`` itself — only its equivalence class.  Worth pinning, because a
    round-trip test that failed here would be testing the wrong thing."""
    transform = T.SoftmaxTransform()
    back = _v(transform.inv(transform(_t(X))))
    offsets = back - X
    assert np.allclose(offsets, offsets[0], atol=1e-9)
    assert not np.allclose(back, X)


# ── lower Cholesky ────────────────────────────────────────────────────────────

LOWER_INPUTS = [
    np.array([[1.0, 0.3], [0.5, 1.0]]),
    np.array([[0.2, -1.0], [-0.7, 2.0]]),
    np.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.3, 0.4, 1.5]]),
    np.array([[30.0, 0.0], [1.0, 25.0]]),
]


@pytest.mark.parametrize("x", LOWER_INPUTS)
def test_lower_cholesky_lands_on_a_positive_diagonal_factor(x):
    y = _v(T.LowerCholeskyTransform()(_t(x)))
    assert np.allclose(np.triu(y, 1), 0.0)
    assert (np.diag(y) > 0).all()


@pytest.mark.parametrize("x", LOWER_INPUTS)
def test_lower_cholesky_round_trips(x):
    """It had no working inverse at all.

    The diagonal was masked out *before* inverting the softplus, so every
    off-diagonal lane computed ``log(exp(0) - 1) = log(0) = -inf`` and
    then ``-inf * 0``, i.e. NaN.  The whole strict lower triangle came
    back NaN — and with it ``log_prob`` for any distribution pushed
    through this transform, since that goes through the inverse.

    The last case has a diagonal of 30: ``exp(z) - 1`` loses its
    significant digits well before that, so ``expm1`` is not decoration.
    """
    transform = T.LowerCholeskyTransform()
    back = _v(transform.inv(transform(_t(x))))
    assert np.isfinite(back).all()
    assert np.allclose(np.tril(back), np.tril(x), atol=1e-8)


def test_lower_cholesky_log_det_is_the_jacobian():
    transform = T.LowerCholeskyTransform()
    dim = 3
    lower = np.tril_indices(dim)
    free = np.array([1.0, 0.5, 2.0, -0.3, 0.4, 1.5])

    def forward(flat):
        matrix = np.zeros((dim, dim))
        matrix[lower] = flat
        return _v(transform(_t(matrix)))[lower]

    matrix = np.zeros((dim, dim))
    matrix[lower] = free
    got = float(_v(transform.log_abs_det_jacobian(_t(matrix), transform(_t(matrix)))).ravel()[0])
    assert np.isclose(got, _numeric_logdet(forward, free), atol=1e-5)


# ── correlation Cholesky ──────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_corr_cholesky_lands_on_a_correlation_factor(dim):
    free = np.random.default_rng(dim).standard_normal(dim * (dim - 1) // 2)
    factor = _v(T.CorrCholeskyTransform(dim)(_t(free)))
    assert np.allclose(np.triu(factor, 1), 0.0)
    corr = factor @ factor.T
    assert np.allclose(np.diag(corr), 1.0, atol=1e-9)
    assert (np.linalg.eigvalsh(corr) > -1e-10).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_corr_cholesky_round_trips(dim):
    transform = T.CorrCholeskyTransform(dim)
    free = np.random.default_rng(dim).standard_normal(dim * (dim - 1) // 2)
    assert np.allclose(_v(transform.inv(transform(_t(free)))), free, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_corr_cholesky_log_det_is_the_jacobian(dim):
    """Against the free (strictly lower) entries, which is the only
    determinant that is defined when domain and codomain have different
    dimensions.

    What stood here was a guess — the comment above it sketched three
    formulas and settled on weighting ``log L_rr`` by ``d - row``.  At
    ``d = 3`` that gives ``-0.5973`` for ``-0.3885``: a wrong density,
    not a wrong constant.
    """
    transform = T.CorrCholeskyTransform(dim)
    lower = np.tril_indices(dim, -1)
    free = np.random.default_rng(dim).standard_normal(dim * (dim - 1) // 2)
    got = float(_v(transform.log_abs_det_jacobian(_t(free), transform(_t(free)))).ravel()[0])
    numeric = _numeric_logdet(lambda z: _v(transform(_t(z)))[lower], free)
    assert np.isclose(got, numeric, atol=1e-5)


# ── the combinators ───────────────────────────────────────────────────────────


def test_reshape_moves_between_shapes_and_back():
    transform = T.ReshapeTransform((2, 3), (6,))
    x = np.arange(6.0).reshape(2, 3)
    y = transform(_t(x))
    assert _v(y).shape == (6,)
    assert np.allclose(_v(transform.inv(y)), x)


def test_independent_sums_the_log_det_over_the_reinterpreted_dims():
    transform = T.IndependentTransform(T.ExpTransform(), 1)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = transform(_t(x))
    ldj = _v(transform.log_abs_det_jacobian(_t(x), y))
    assert ldj.shape == (2,)  # one per batch element, not per element
    assert np.allclose(ldj, x.sum(axis=1))
    assert np.allclose(_v(transform.inv(y)), x, atol=1e-9)


def test_cat_applies_a_different_transform_to_each_slice():
    transform = T.CatTransform(
        [T.ExpTransform(), T.AffineTransform(1.0, 2.0)], dim=0, lengths=[2, 2]
    )
    y = _v(transform(_t(X)))
    assert np.allclose(y[:2], np.exp(X[:2]))
    assert np.allclose(y[2:], 1.0 + 2.0 * X[2:])
    assert np.allclose(_v(transform.inv(_t(y))), X, atol=1e-9)


def test_stack_applies_a_different_transform_to_each_index():
    transform = T.StackTransform([T.ExpTransform(), T.SigmoidTransform()], dim=0)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = _v(transform(_t(x)))
    assert np.allclose(y[0], np.exp(x[0]))
    assert np.allclose(y[1], 1.0 / (1.0 + np.exp(-x[1])))
    assert np.allclose(_v(transform.inv(_t(y))), x, atol=1e-9)


@pytest.mark.parametrize("direction", ["forward", "inverse", "log_det"])
def test_stack_refuses_a_slice_count_other_than_its_transforms(direction):
    """Every direction counts the slices, not only the forward.

    The inverse and the log-det paired slices with transforms by ``zip``,
    which raised its own error once it turned strict and before that
    dropped the extra slices.
    """
    transform = T.StackTransform([T.ExpTransform(), T.TanhTransform()], dim=-1)
    v = _t(np.array([0.5, 0.2, 0.1]))
    calls = {
        "forward": lambda: transform(v),
        "inverse": lambda: transform.inv(v),
        "log_det": lambda: transform.log_abs_det_jacobian(v, v),
    }
    with pytest.raises(ValueError, match="got 3 slices but 2 transforms"):
        calls[direction]()


@pytest.mark.parametrize("lengths", [[1, 1, 1], [2]])
def test_cat_refuses_lengths_that_are_not_one_per_transform(lengths):
    with pytest.raises(ValueError, match=f"got {len(lengths)} lengths but 2"):
        T.CatTransform([T.ExpTransform(), T.TanhTransform()], lengths=lengths)


def test_compose_is_the_transforms_in_order():
    composed = T.ComposeTransform([T.ExpTransform(), T.AffineTransform(1.0, 2.0)])
    assert np.allclose(_v(composed(_t(X))), 1.0 + 2.0 * np.exp(X))


def test_compose_adds_the_log_dets():
    first, second = T.ExpTransform(), T.AffineTransform(1.0, 2.0)
    composed = T.ComposeTransform([first, second])
    mid = first(_t(X))
    want = _v(first.log_abs_det_jacobian(_t(X), mid)) + _v(
        second.log_abs_det_jacobian(mid, second(mid))
    )
    assert np.allclose(_v(composed.log_abs_det_jacobian(_t(X), composed(_t(X)))), want)


def test_the_cdf_transform_lands_on_the_unit_interval():
    transform = T.CumulativeDistributionTransform(D.Normal(_t(0.0), _t(1.0)))
    y = _v(transform(_t(X)))
    assert ((y > 0.0) & (y < 1.0)).all()
    assert np.allclose(_v(transform.inv(_t(y))), X, atol=1e-6)


def test_the_cdf_transform_of_a_variable_is_uniform():
    """The probability integral transform, which is the point of it."""
    lucid.manual_seed(0)
    normal = D.Normal(_t(0.0), _t(1.0))
    draws = _v(normal.sample((20000,))).ravel()
    pushed = _v(T.CumulativeDistributionTransform(normal)(_t(draws)))
    counts, _ = np.histogram(pushed, bins=10, range=(0.0, 1.0))
    assert np.abs(counts / counts.sum() - 0.1).max() < 0.01


# ── TransformedDistribution ───────────────────────────────────────────────────


def test_an_exponentiated_normal_is_log_normal():
    transform = T.TransformedDistribution(D.Normal(_t(0.0), _t(1.0)), T.ExpTransform())
    lucid.manual_seed(1)
    draws = _v(transform.sample((5000,)))
    assert (draws > 0).all()
    got = float(_v(transform.log_prob(_t(2.0))).ravel()[0])
    want = -math.log(2.0) - 0.5 * math.log(2 * math.pi) - 0.5 * math.log(2.0) ** 2
    assert np.isclose(got, want, atol=1e-8)


def test_a_chain_of_transforms_pushes_the_density_through_each():
    base = D.Normal(_t(0.0), _t(1.0))
    chain = T.ComposeTransform([T.ExpTransform(), T.AffineTransform(0.0, 3.0)])
    pushed = T.TransformedDistribution(base, chain)
    x = 0.4
    y = float(_v(chain(_t(x))).ravel()[0])
    want = float(_v(base.log_prob(_t(x))).ravel()[0]) - float(
        _v(chain.log_abs_det_jacobian(_t(x), _t(y))).ravel()[0]
    )
    assert np.isclose(float(_v(pushed.log_prob(_t(y))).ravel()[0]), want, atol=1e-8)


# ── the non-injective ones ────────────────────────────────────────────────────


def test_abs_is_forward_only_and_reports_a_zero_log_det():
    """Recorded rather than endorsed.

    ``abs`` is not injective, so it has no inverse and no determinant.
    The reference refuses ``log_abs_det_jacobian`` outright; Lucid
    answers ``0``, which is the right value on each of the two branches
    but reads as "this is a volume-preserving bijection", which it is
    not.  Pinned so the choice is at least visible.
    """
    transform = T.AbsTransform()
    assert np.allclose(_v(transform(_t(X))), np.abs(X))
    assert np.allclose(_v(transform.log_abs_det_jacobian(_t(X), transform(_t(X)))), 0.0)


# ── domain / codomain constraints, support, and boundary clamping (LCD-397) ──


def _transform_constraint_cases(ref):
    rt = ref.distributions.transforms
    return [
        (
            "ExpTransform",
            T.ExpTransform(),
            rt.ExpTransform(),
            [-1.0, 0.0, 2.5],
            [-1.0, 0.0, 0.5, 2.5],
        ),
        (
            "SigmoidTransform",
            T.SigmoidTransform(),
            rt.SigmoidTransform(),
            [-3.0, 0.0, 3.0],
            [-0.1, 0.0, 0.5, 1.0, 1.1],
        ),
        (
            "TanhTransform",
            T.TanhTransform(),
            rt.TanhTransform(),
            [-3.0, 0.0, 3.0],
            [-1.5, -1.0, 0.0, 1.0, 1.5],
        ),
        (
            "AffineTransform",
            T.AffineTransform(1.0, 2.0),
            rt.AffineTransform(1.0, 2.0),
            [-2.0, 0.0, 2.0],
            [-2.0, 0.0, 2.0],
        ),
        (
            "PowerTransform",
            T.PowerTransform(_t(2.0)),
            rt.PowerTransform(ref.tensor(2.0, dtype=ref.float64)),
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
        ),
        (
            "SoftmaxTransform",
            T.SoftmaxTransform(),
            rt.SoftmaxTransform(),
            [[0.0, 1.0, -1.0], [2.0, 3.0, 4.0]],
            [[0.2, 0.3, 0.5], [-0.1, 0.6, 0.5], [0.5, 0.5, 0.5]],
        ),
        (
            "StickBreakingTransform",
            T.StickBreakingTransform(),
            rt.StickBreakingTransform(),
            [[0.0, 1.0], [-1.0, 2.0]],
            [[0.2, 0.3, 0.5], [-0.1, 0.6, 0.5], [0.5, 0.5, 0.5]],
        ),
        (
            "LowerCholeskyTransform",
            T.LowerCholeskyTransform(),
            rt.LowerCholeskyTransform(),
            [[[1.0, 0.0], [0.5, 2.0]], [[1.0, 2.0], [0.5, -1.0]]],
            [[[1.0, 0.0], [0.5, 2.0]], [[1.0, 2.0], [0.5, 2.0]], [[1.0, 0.0], [0.5, -1.0]]],
        ),
        (
            "CorrCholeskyTransform",
            T.CorrCholeskyTransform(3),
            rt.CorrCholeskyTransform(),
            [[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]],
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
        ),
        ("AbsTransform", T.AbsTransform(), rt.AbsTransform(), [-2.0, 0.0, 2.0], [-1.0, 0.0, 2.0]),
        (
            "IndependentTransform",
            T.IndependentTransform(T.ExpTransform(), 1),
            rt.IndependentTransform(rt.ExpTransform(), 1),
            [[-1.0, 0.5], [0.0, 2.0]],
            [[0.5, 2.0], [-1.0, 2.0], [0.0, 1.0]],
        ),
        (
            "ReshapeTransform",
            T.ReshapeTransform((2, 3), (6,)),
            rt.ReshapeTransform((2, 3), (6,)),
            [np.zeros((2, 3)).tolist(), np.ones((2, 3)).tolist()],
            [np.zeros((6,)).tolist(), np.ones((6,)).tolist()],
        ),
        (
            "CumulativeDistributionTransform",
            T.CumulativeDistributionTransform(D.Exponential(_t(1.0))),
            rt.CumulativeDistributionTransform(
                ref.distributions.Exponential(ref.tensor(1.0, dtype=ref.float64))
            ),
            [-1.0, 0.0, 1.5],
            [-0.5, 0.0, 0.5, 1.0, 1.5],
        ),
        (
            "StackTransform",
            T.StackTransform([T.ExpTransform(), T.SigmoidTransform()], dim=-1),
            rt.StackTransform([rt.ExpTransform(), rt.SigmoidTransform()], dim=-1),
            [[-1.0, 0.0], [2.0, -2.0]],
            [[1.5, 0.5], [-1.0, 0.5], [1.5, 1.5], [0.0, 0.5]],
        ),
        (
            "CatTransform",
            T.CatTransform([T.ExpTransform(), T.SigmoidTransform()], dim=-1, lengths=[2, 1]),
            rt.CatTransform([rt.ExpTransform(), rt.SigmoidTransform()], dim=-1, lengths=[2, 1]),
            [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]],
            [[1.5, 0.5, 0.5], [-1.0, 0.5, 0.5], [1.5, 0.5, 1.5]],
        ),
        (
            "ComposeTransform",
            T.ComposeTransform([T.ExpTransform(), T.IndependentTransform(T.SigmoidTransform(), 1)]),
            rt.ComposeTransform(
                [rt.ExpTransform(), rt.IndependentTransform(rt.SigmoidTransform(), 1)]
            ),
            [[-1.0, 0.0], [1.0, 2.0]],
            [[0.2, 0.8], [-0.1, 0.8], [0.2, 1.2]],
        ),
        (
            "InverseTransform(ExpTransform)",
            T.ExpTransform().inv,
            rt.ExpTransform().inv,
            [-1.0, 0.0, 0.5, 2.5],
            [-1.0, 0.0, 2.5],
        ),
    ]


def test_every_transform_declares_matching_domain_and_codomain(ref):
    for name, ours, theirs, dom_vals, cod_vals in _transform_constraint_cases(ref):
        assert ours.domain.event_dim == theirs.domain.event_dim, f"{name} domain.event_dim"
        assert ours.domain.is_discrete == theirs.domain.is_discrete, f"{name} domain.is_discrete"
        assert ours.codomain.event_dim == theirs.codomain.event_dim, f"{name} codomain.event_dim"
        assert (
            ours.codomain.is_discrete == theirs.codomain.is_discrete
        ), f"{name} codomain.is_discrete"

        x_l = _t(dom_vals)
        x_r = ref.tensor(np.asarray(dom_vals, dtype=np.float64))
        assert np.array_equal(
            _v(ours.domain.check(x_l)), theirs.domain.check(x_r).numpy()
        ), f"{name} domain.check mismatch"

        y_l = _t(cod_vals)
        y_r = ref.tensor(np.asarray(cod_vals, dtype=np.float64))
        assert np.array_equal(
            _v(ours.codomain.check(y_l)), theirs.codomain.check(y_r).numpy()
        ), f"{name} codomain.check mismatch"


def test_transformed_distribution_reexport_and_support_validation(ref):
    from lucid.distributions.transformed_distribution import TransformedDistribution as DirectTD

    assert DirectTD is T.TransformedDistribution
    assert DirectTD is D.TransformedDistribution

    dist = DirectTD(
        D.Normal(_t(0.0), _t(1.0)),
        [T.AffineTransform(0.0, 1.0), T.ExpTransform()],
        validate_args=True,
    )
    ref_dist = ref.distributions.TransformedDistribution(
        ref.distributions.Normal(ref.tensor(0.0), ref.tensor(1.0)),
        [
            ref.distributions.transforms.AffineTransform(0.0, 1.0),
            ref.distributions.transforms.ExpTransform(),
        ],
        validate_args=True,
    )
    assert dist.support is not None
    assert dist.support.event_dim == ref_dist.support.event_dim
    for bad in (-1.0, 0.0):
        with pytest.raises(ValueError):
            dist.log_prob(_t(bad))
    assert np.isclose(
        float(_v(dist.log_prob(_t(1.5))).ravel()[0]),
        float(ref_dist.log_prob(ref.tensor(1.5)).item()),
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sigmoid_transform_inverse_clamps_to_finfo_bounds(dtype_name, ref):
    l_dtype = getattr(lucid, dtype_name)
    r_dtype = getattr(ref, dtype_name)
    pts_l = lucid.tensor([0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0], dtype=l_dtype)
    pts_r = ref.tensor([0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0], dtype=r_dtype)
    got = _v(T.SigmoidTransform()._inverse(pts_l))
    want = ref.distributions.transforms.SigmoidTransform()._inverse(pts_r).numpy()
    assert np.isfinite(got).all()
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6)


# ── the rightmost-dims sum behind every per-event reduction (LCD-471) ────────
#
# ``TransformedDistribution`` and the KL registrations each kept their own
# copy, and the two disagreed.  The axis-sum copy raised ``IndexError`` when
# there were more dims to sum than the tensor had; the reshape copy raised on
# an empty batch.  There is one owner now, held to the reference wherever the
# reference returns a value.

_RIGHTMOST_CASES = [
    # (id, shape, dim, permutation applied before the sum)
    ("dim-zero", (2, 3, 4), 0, None),
    ("one-dim", (2, 3, 4), 1, None),
    ("dim-equals-ndim", (2, 3, 4), 3, None),
    ("dim-exceeds-ndim", (2, 3, 4), 5, None),
    ("non-contiguous", (2, 3, 4), 2, (2, 0, 1)),
    ("zero-dim", (), 1, None),
    # The Wishart KL's degrees-of-freedom term: 0-dim, reduced by a rank-2 event.
    ("zero-dim-wishart-df", (), 2, None),
    ("empty-summed-dim", (3, 0), 1, None),
]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize(
    ("shape", "dim", "perm"),
    [case[1:] for case in _RIGHTMOST_CASES],
    ids=[case[0] for case in _RIGHTMOST_CASES],
)
def test_sum_rightmost_matches_the_reference(ref, device, shape, dim, perm):
    data = np.linspace(-1.0, 2.0, num=math.prod(shape)).reshape(shape)
    data = data.astype(np.float32)
    ours = lucid.tensor(data).to(device)
    theirs = ref.tensor(data)
    if perm is not None:
        ours, theirs = ours.permute(*perm), theirs.permute(*perm)
        # A Metal permute may come back materialised; the CPU one is a strided view.
        assert device == "metal" or not ours.is_contiguous()
    got = _sum_rightmost(ours, dim)
    want = ref.distributions.utils._sum_rightmost(theirs, dim)
    assert got.device.type == device
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(got.to("cpu").numpy(), want.numpy(), rtol=1e-5, atol=1e-6)
    if dim == 0:
        assert got is ours


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize(("shape", "dim", "want"), [((0, 3), 1, (0,)), ((2, 0, 3), 1, (2, 0))])
def test_sum_rightmost_sums_an_empty_batch_to_an_empty_batch(device, shape, dim, want):
    """The documented departure from the reference.

    The reference flattens with ``reshape(..., -1)``, which cannot infer the
    size when a kept dim is 0, so it raises on these inputs.
    """
    got = _sum_rightmost(lucid.zeros(*shape, device=device), dim)
    assert tuple(got.shape) == want


def test_both_former_copies_are_the_one_owner():
    assert transformed_distribution._sum_rightmost is _sum_rightmost
    assert kl_cross._sum_rightmost is _sum_rightmost


def test_a_log_det_of_lower_rank_than_the_event_is_summed_whole(ref):
    """``TransformedDistribution.log_prob`` raised ``IndexError`` here.

    A transform may report a constant volume change once rather than once per
    element.  Its 0-dim log-det then has fewer dims than the rank-2 event it
    is summed over, and the reference sums it whole.
    """
    log_scale = math.log(2.0)

    class _ConstantLogDet(T.AffineTransform):
        def log_abs_det_jacobian(self, x, y):
            return _t(log_scale)

    class _RefConstantLogDet(ref.distributions.transforms.AffineTransform):
        def log_abs_det_jacobian(self, x, y):
            return ref.tensor(log_scale, dtype=ref.float64)

    rd = ref.distributions
    value = np.linspace(-1.0, 1.0, 12).reshape(4, 3)
    ours = T.TransformedDistribution(
        D.Independent(D.Normal(_t(np.zeros((4, 3))), _t(np.ones((4, 3)))), 2),
        [_ConstantLogDet(0.0, 2.0)],
    )
    theirs = rd.TransformedDistribution(
        rd.Independent(
            rd.Normal(ref.zeros(4, 3, dtype=ref.float64), ref.ones(4, 3, dtype=ref.float64)), 2
        ),
        [_RefConstantLogDet(0.0, 2.0)],
    )
    got = ours.log_prob(_t(value))
    want = theirs.log_prob(ref.tensor(value))
    assert tuple(got.shape) == tuple(want.shape) == ()
    assert np.isclose(float(got.item()), float(want.item()), atol=1e-8)


def test_an_empty_batch_has_an_empty_transformed_log_prob():
    pushed = T.TransformedDistribution(
        D.Independent(D.Normal(_t(np.zeros(3)), _t(np.ones(3))), 1), [T.ExpTransform()]
    )
    assert tuple(pushed.log_prob(_t(np.ones((0, 3)))).shape) == (0,)


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("reinterpreted_ndims", [1, 2])
def test_independent_transform_with_event_dim_inner_matches_reference(
    ref, device, reinterpreted_ndims
):
    """LCD-482: ``IndependentTransform.log_abs_det_jacobian`` sums over
    ``reinterpreted_batch_ndims`` (not ``reinterpreted_batch_ndims - inner.event_dim``),
    because the inner transform has already reduced its own event dims."""
    rt = ref.distributions.transforms
    raw = np.linspace(-0.8, 0.8, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    x = lucid.tensor(raw, device=device)
    rx = ref.tensor(raw)

    ours = T.IndependentTransform(T.StickBreakingTransform(), reinterpreted_ndims)
    theirs = rt.IndependentTransform(rt.StickBreakingTransform(), reinterpreted_ndims)
    y = ours(x)
    ry = theirs(rx)
    got = ours.log_abs_det_jacobian(x, y)
    want = theirs.log_abs_det_jacobian(rx, ry)

    assert got.device.type == device
    assert tuple(got.shape) == tuple(want.shape) == (2, 3)[: 2 - reinterpreted_ndims]
    np.testing.assert_allclose(got.to("cpu").numpy(), want.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize(("dim", "lengths"), [(-1, [2, 2]), (1, [2, 2]), (0, [1, 2]), (-2, [1, 2])])
def test_cat_transform_mixed_event_dim_matches_reference(ref, device, dim, lengths):
    """LCD-507: ``CatTransform.log_abs_det_jacobian`` reduces partitions with
    smaller ``event_dim`` via ``_sum_rightmost`` and concatenates or sums
    according to whether ``dim`` falls inside ``self.event_dim``."""
    rt = ref.distributions.transforms
    raw = np.linspace(-0.8, 0.8, 3 * 4, dtype=np.float32).reshape(3, 4)
    x = lucid.tensor(raw, device=device)
    rx = ref.tensor(raw)

    ours = T.CatTransform(
        [T.ExpTransform(), T.IndependentTransform(T.ExpTransform(), 1)], dim=dim, lengths=lengths
    )
    theirs = rt.CatTransform(
        [rt.ExpTransform(), rt.IndependentTransform(rt.ExpTransform(), 1)], dim=dim, lengths=lengths
    )
    y = ours(x)
    ry = theirs(rx)
    got = ours.log_abs_det_jacobian(x, y)
    want = theirs.log_abs_det_jacobian(rx, ry)

    assert got.device.type == device
    assert tuple(got.shape) == tuple(want.shape) == (3,)
    np.testing.assert_allclose(got.to("cpu").numpy(), want.numpy(), rtol=1e-5, atol=1e-6)
