"""``linalg.norm`` dispatches by rank and axes, as its docstring says.

It used to forward every argument straight into the engine's flat
p-norm, so ``ord`` was read as a *vector* order over the flattened input
whatever the rank.  On a matrix that is a different function under the
same name, and none of it raised:

    ord=1     21     where the matrix 1-norm is 9   (max column sum)
    ord=2     9.539  where the spectral norm is 9.508
    ord=inf   6      where the matrix inf-norm is 15 (max row sum)

``ord=2`` is the dangerous one: Frobenius and spectral agree to three
significant figures on well-conditioned matrices and separate exactly
where the answer starts to matter.

Expected values here are the definitions written out, not readings taken
from an implementation.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.linalg as L

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
V = np.array([1.0, -2.0, 3.0])
B = np.arange(1.0, 13.0).reshape(2, 2, 3)


def _v(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.numpy())


# ── matrix orders, on a matrix ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ord_,expected",
    [
        # max absolute column sum: columns are (1,4), (2,5), (3,6) -> 5, 7, 9
        (1, 9.0),
        (-1, 5.0),
        # max absolute row sum: rows sum to 6 and 15
        (math.inf, 15.0),
        (-math.inf, 6.0),
        # Frobenius: sqrt(1+4+9+16+25+36)
        ("fro", math.sqrt(91.0)),
    ],
)
def test_matrix_orders_are_matrix_norms(ord_, expected) -> None:
    got = float(_v(L.norm(lucid.tensor(A), ord=ord_)).ravel()[0])
    assert abs(got - expected) < 1e-10, (got, expected)


def test_ord_two_on_a_matrix_is_spectral_not_frobenius() -> None:
    """The two differ by ~0.3% here, which is why this went unnoticed."""
    spectral = float(np.linalg.svd(A, compute_uv=False)[0])
    frobenius = float(np.sqrt((A**2).sum()))
    assert abs(spectral - frobenius) > 1e-3

    got = float(_v(L.norm(lucid.tensor(A), ord=2)).ravel()[0])
    assert abs(got - spectral) < 1e-8, (got, spectral)
    assert abs(got - frobenius) > 1e-3


def test_nuclear_norm_is_the_sum_of_singular_values() -> None:
    expected = float(np.linalg.svd(A, compute_uv=False).sum())
    got = float(_v(L.norm(lucid.tensor(A), ord="nuc")).ravel()[0])
    assert abs(got - expected) < 1e-8


# ── vector orders, on a vector ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ord_,expected",
    [
        (None, math.sqrt(14.0)),
        (1, 6.0),
        (2, math.sqrt(14.0)),
        (3, 36.0 ** (1.0 / 3.0)),
        (math.inf, 3.0),
        (-math.inf, 1.0),
        (0, 3.0),  # a count, not a norm
    ],
)
def test_vector_orders(ord_, expected) -> None:
    got = float(_v(L.norm(lucid.tensor(V), ord=ord_)).ravel()[0])
    assert abs(got - expected) < 1e-10, (got, expected)


def test_ord_zero_is_a_count_and_the_rescaling_must_not_touch_it() -> None:
    """Counting is scale-invariant, so multiplying the scale back is wrong.

    The overflow rescaling divides by the largest magnitude and
    multiplies it back at the end, which is exact for every order that is
    absolutely homogeneous.  ``ord=0`` is not: it answered 9 for a vector
    with three non-zero entries, the count times the largest entry.
    """
    assert float(_v(L.norm(lucid.tensor(V), ord=0)).ravel()[0]) == 3.0
    big = lucid.tensor(np.array([1e100, 0.0, -2.0, 0.0]))
    assert float(_v(L.norm(big, ord=0)).ravel()[0]) == 2.0
    assert float(_v(L.norm(lucid.tensor(np.zeros(4)), ord=0)).ravel()[0]) == 0.0


def test_ord_zero_returns_a_float() -> None:
    """A norm whose dtype depends on its order breaks whatever divides by it."""
    out = L.norm(lucid.tensor(V), ord=0)
    assert "float" in str(out.dtype), out.dtype


# ── dim decides, and overrules the rank ───────────────────────────────────────


def test_one_axis_is_a_vector_norm_even_on_a_matrix() -> None:
    got = _v(L.norm(lucid.tensor(A), ord=1, dim=1))
    assert np.allclose(got, [6.0, 15.0])  # per-row, not the matrix 1-norm of 9


def test_two_axes_are_a_matrix_norm_even_on_a_batch() -> None:
    got = _v(L.norm(lucid.tensor(B), ord=1, dim=(1, 2)))
    assert np.allclose(got, [9.0, 21.0])


@pytest.mark.parametrize("form", [1, [1], (1,)])
def test_a_single_axis_may_be_written_three_ways(form) -> None:
    assert np.allclose(_v(L.norm(lucid.tensor(A), ord=1, dim=form)), [6.0, 15.0])


def test_no_ord_and_no_dim_flattens_at_any_rank() -> None:
    for arr in (V, A, B):
        expected = float(np.sqrt((arr**2).sum()))
        got = float(_v(L.norm(lucid.tensor(arr))).ravel()[0])
        assert abs(got - expected) < 1e-8, (arr.shape, got, expected)


# ── shapes ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,shape",
    [
        ({}, ()),
        ({"keepdim": True}, (1, 1, 1)),
        ({"ord": 1, "dim": (1, 2)}, (2,)),
        ({"ord": 1, "dim": (1, 2), "keepdim": True}, (2, 1, 1)),
        ({"ord": "nuc", "dim": (0, 1)}, (3,)),
        ({"ord": "nuc", "dim": (0, 1), "keepdim": True}, (1, 1, 3)),
        ({"ord": 2, "dim": (2, 0)}, (2,)),
        ({"ord": 2, "dim": 1}, (2, 3)),
    ],
)
def test_result_shapes(kwargs, shape) -> None:
    assert L.norm(lucid.tensor(B), **kwargs).shape == shape


def test_singular_value_orders_survive_a_transposed_axis_pair() -> None:
    """``dim=(2, 0)`` names the plane in the other order; the SVD is the same."""
    got = _v(L.norm(lucid.tensor(B), ord=2, dim=(2, 0)))
    expected = [float(np.linalg.svd(B[:, i, :].T, compute_uv=False)[0]) for i in range(2)]
    assert np.allclose(got, expected)


# ── refusals ──────────────────────────────────────────────────────────────────


def test_an_order_without_axes_needs_a_vector_or_one_matrix() -> None:
    """A batch leaves it ambiguous; picking one reading silently is worse."""
    with pytest.raises(ValueError, match="1-D or 2-D"):
        L.norm(lucid.tensor(B), ord=2)


def test_a_matrix_order_is_refused_on_a_vector() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        L.norm(lucid.tensor(V), ord="fro")


def test_a_matrix_order_needs_two_axes() -> None:
    with pytest.raises(ValueError, match="pair of axes"):
        L.norm(lucid.tensor(A), ord="fro", dim=1)


@pytest.mark.parametrize("ord_", [0, 3, 0.5])
def test_matrix_norm_refuses_orders_it_has_no_definition_for(ord_) -> None:
    with pytest.raises(ValueError, match="unsupported ord"):
        L.matrix_norm(lucid.tensor(A), ord=ord_)


def test_three_axes_is_neither_norm() -> None:
    with pytest.raises(ValueError, match="one or two axes"):
        L.norm(lucid.tensor(B), ord=1, dim=(0, 1, 2))


# ── overflow, which the rescaling exists for ──────────────────────────────────


@pytest.mark.parametrize(
    "arr,expected",
    [
        (np.array([1e200, 1e200]), math.sqrt(2.0) * 1e200),
        (np.array([1e-200, 1e-200]), math.sqrt(2.0) * 1e-200),
        (np.full((2, 2), 1e200), 2e200),
    ],
)
def test_a_representable_norm_is_computed(arr, expected) -> None:
    got = float(_v(L.norm(lucid.tensor(arr))).ravel()[0])
    assert np.isfinite(got)
    assert abs(got - expected) / expected < 1e-12, (got, expected)


@pytest.mark.parametrize(
    "arr,check",
    [
        (np.array([np.inf, 1.0]), lambda g: g == math.inf),
        (np.array([-np.inf, 1.0]), lambda g: g == math.inf),
        (np.array([np.nan, 1.0]), lambda g: math.isnan(g)),
        (np.zeros(3), lambda g: g == 0.0),
    ],
)
def test_rescaling_does_not_manufacture_nan(arr, check) -> None:
    """The scale is the largest magnitude, so an infinity makes it inf.

    Dividing by it turns every entry into NaN or 0 and the answer with
    it — ``norm([inf, 1])`` reported NaN where it is plainly inf.  An
    all-zero input divides by 0 the same way.  Both fall back to an
    unscaled evaluation, which is already right there.
    """
    assert check(float(_v(L.norm(lucid.tensor(arr))).ravel()[0]))
