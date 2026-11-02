"""Transform log-det availability and number-parameter precision against the reference (LCD-818).

Two things LCD-816 left in ``lucid.distributions.transforms``:

* ``SoftmaxTransform.log_abs_det_jacobian`` answered ``sum(log y)`` — a
  "pseudo-Jacobian" with no determinant behind it — and ``AbsTransform`` a flat
  ``0``, where the reference refuses both.  Which transforms refuse and which
  compute is held to the reference here, wrappers included (a composition or an
  inverse is only as defined as its parts).
* ``AffineTransform`` kept a Python-number ``loc`` / ``scale`` as a float32
  scalar, so ``AffineTransform(-1.0, 0.3)`` on float64 input was off by ~5e-9.
  A number operand now takes the dtype of the tensor it meets.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.distributions.transforms as T

pytestmark = pytest.mark.parity

_VEC = [0.4, -0.3, 1.2]
_POS = [0.4, 0.3, 1.2]
_SIMPLEX = [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]
_ROWS = [[0.4, -0.3, 1.2], [-0.8, 0.5, 0.0]]
_TRIL = [[[0.4, 0.0], [-0.3, 1.2]], [[0.9, 0.0], [0.1, -0.5]]]


def _lt(value: Any, device: str = "cpu", dtype: Any = np.float32) -> lucid.Tensor:
    return lucid.tensor(np.asarray(value, dtype=dtype)).to(device)


def _rt(ref: Any, value: Any, dtype: Any = np.float32) -> Any:
    return ref.tensor(np.asarray(value, dtype=dtype))


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)


def _outcome(fn: Callable[[], Any]) -> tuple[str, Any]:
    """``("ok", value)``, or the name of the exception class the call raised."""
    try:
        out = fn()
    except (NotImplementedError, ValueError, TypeError) as exc:
        return type(exc).__name__, None
    return "ok", _arr(out)


# ── Which transforms have a log-det ──────────────────────────────────────────


def _cases(ref: Any) -> dict[str, tuple[Callable[[], Any], Callable[[], Any], Any]]:
    """Name -> (Lucid factory, reference factory, an input in the transform's domain)."""
    RT = ref.distributions.transforms
    reg = D.constraint_registry
    return {
        "Exp": (T.ExpTransform, RT.ExpTransform, _VEC),
        "Sigmoid": (T.SigmoidTransform, RT.SigmoidTransform, _VEC),
        "Tanh": (T.TanhTransform, RT.TanhTransform, _VEC),
        "Softplus": (reg.SoftplusTransform, RT.SoftplusTransform, _VEC),
        "Affine(+)": (
            lambda: T.AffineTransform(0.5, 2.0),
            lambda: RT.AffineTransform(0.5, 2.0),
            _VEC,
        ),
        "Affine(-)": (
            lambda: T.AffineTransform(0.5, -2.0),
            lambda: RT.AffineTransform(0.5, -2.0),
            _VEC,
        ),
        "Affine(tensor)": (
            lambda: T.AffineTransform(
                lucid.tensor([0.5, 0.0, 1.0]), lucid.tensor([2.0, -1.0, 3.0])
            ),
            lambda: RT.AffineTransform(ref.tensor([0.5, 0.0, 1.0]), ref.tensor([2.0, -1.0, 3.0])),
            _VEC,
        ),
        "Power": (lambda: T.PowerTransform(2.0), lambda: RT.PowerTransform(2.0), _POS),
        "Softmax": (T.SoftmaxTransform, RT.SoftmaxTransform, _ROWS),
        "StickBreaking": (T.StickBreakingTransform, RT.StickBreakingTransform, _ROWS),
        "CorrCholesky": (
            lambda: T.CorrCholeskyTransform(3),
            RT.CorrCholeskyTransform,
            [[0.4, -0.3, 0.2], [0.1, 0.5, -0.6]],
        ),
        "LowerCholesky": (T.LowerCholeskyTransform, RT.LowerCholeskyTransform, _TRIL),
        "PositiveDefinite": (reg.PositiveDefiniteTransform, RT.PositiveDefiniteTransform, _TRIL),
        "Abs": (T.AbsTransform, RT.AbsTransform, _VEC),
        "CDF": (
            lambda: T.CumulativeDistributionTransform(D.Normal(0.0, 1.0)),
            lambda: RT.CumulativeDistributionTransform(ref.distributions.Normal(0.0, 1.0)),
            _VEC,
        ),
        "Reshape": (
            lambda: T.ReshapeTransform((3,), (3, 1)),
            lambda: RT.ReshapeTransform((3,), (3, 1)),
            _ROWS,
        ),
        "Independent(Exp)": (
            lambda: T.IndependentTransform(T.ExpTransform(), 1),
            lambda: RT.IndependentTransform(RT.ExpTransform(), 1),
            _ROWS,
        ),
        "Independent(Softmax)": (
            lambda: T.IndependentTransform(T.SoftmaxTransform(), 1),
            lambda: RT.IndependentTransform(RT.SoftmaxTransform(), 1),
            _ROWS,
        ),
        "Independent(Abs)": (
            lambda: T.IndependentTransform(T.AbsTransform(), 1),
            lambda: RT.IndependentTransform(RT.AbsTransform(), 1),
            _ROWS,
        ),
        "Compose(Exp,Affine)": (
            lambda: T.ComposeTransform([T.ExpTransform(), T.AffineTransform(0.0, -1.0)]),
            lambda: RT.ComposeTransform([RT.ExpTransform(), RT.AffineTransform(0.0, -1.0)]),
            _VEC,
        ),
        "Compose(Exp,Softmax)": (
            lambda: T.ComposeTransform([T.ExpTransform(), T.SoftmaxTransform()]),
            lambda: RT.ComposeTransform([RT.ExpTransform(), RT.SoftmaxTransform()]),
            _ROWS,
        ),
        "Compose(Abs,Exp)": (
            lambda: T.ComposeTransform([T.AbsTransform(), T.ExpTransform()]),
            lambda: RT.ComposeTransform([RT.AbsTransform(), RT.ExpTransform()]),
            _VEC,
        ),
        "Inverse(Exp)": (lambda: T.ExpTransform().inv, lambda: RT.ExpTransform().inv, _POS),
        "Inverse(Softmax)": (
            lambda: T.SoftmaxTransform().inv,
            lambda: RT.SoftmaxTransform().inv,
            _SIMPLEX,
        ),
        "Inverse(Abs)": (lambda: T.AbsTransform().inv, lambda: RT.AbsTransform().inv, _POS),
        "Cat(Exp,Abs)": (
            lambda: reg.CatTransform([T.ExpTransform(), T.AbsTransform()], 0),
            lambda: RT.CatTransform([RT.ExpTransform(), RT.AbsTransform()], 0),
            [0.4, -0.3],
        ),
        "Stack(Exp,Exp)": (
            lambda: reg.StackTransform([T.ExpTransform(), T.ExpTransform()], 0),
            lambda: RT.StackTransform([RT.ExpTransform(), RT.ExpTransform()], 0),
            [[0.4, -0.3], [0.1, 0.9]],
        ),
        "Stack(Exp,Softmax)": (
            lambda: reg.StackTransform([T.ExpTransform(), T.SoftmaxTransform()], 0),
            lambda: RT.StackTransform([RT.ExpTransform(), RT.SoftmaxTransform()], 0),
            [[0.4, -0.3], [0.1, 0.9]],
        ),
    }


# Where Lucid answers and the reference refuses.  Both are real determinants
# (``test_matrix_transform_log_det_is_the_jacobian`` checks them against a
# numerical Jacobian), kept since LCD-816 rather than removed for parity's
# sake; the set is pinned so a third one cannot join unnoticed.
_LUCID_ONLY = {"LowerCholesky", "PositiveDefinite"}
_NAMES = [
    "Exp",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Affine(+)",
    "Affine(-)",
    "Affine(tensor)",
    "Power",
    "Softmax",
    "StickBreaking",
    "CorrCholesky",
    "LowerCholesky",
    "PositiveDefinite",
    "Abs",
    "CDF",
    "Reshape",
    "Independent(Exp)",
    "Independent(Softmax)",
    "Independent(Abs)",
    "Compose(Exp,Affine)",
    "Compose(Exp,Softmax)",
    "Compose(Abs,Exp)",
    "Inverse(Exp)",
    "Inverse(Softmax)",
    "Inverse(Abs)",
    "Cat(Exp,Abs)",
    "Stack(Exp,Exp)",
    "Stack(Exp,Softmax)",
]


def test_the_matrix_lists_every_case(ref: Any) -> None:
    assert _NAMES == list(_cases(ref))


@pytest.mark.parametrize("name", _NAMES)
def test_log_det_availability_matches_reference(name: str, ref: Any) -> None:
    ours_f, theirs_f, x = _cases(ref)[name]
    ours, theirs = ours_f(), theirs_f()
    xl, xr = _lt(x), _rt(ref, x)
    got = _outcome(lambda: ours.log_abs_det_jacobian(xl, ours(xl)))
    want = _outcome(lambda: theirs.log_abs_det_jacobian(xr, theirs(xr)))
    if name in _LUCID_ONLY:
        assert (got[0], want[0]) == ("ok", "NotImplementedError"), name
        return
    assert got[0] == want[0], f"{name}: Lucid {got[0]}, the reference {want[0]}"
    if want[0] == "ok":
        assert got[1].shape == want[1].shape, name
        np.testing.assert_allclose(got[1], want[1], rtol=3e-4, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("cls", ["SoftmaxTransform", "AbsTransform"])
def test_refusing_log_det_is_the_base_transforms_with_the_class_name(cls: str) -> None:
    transform = getattr(T, cls)()
    x = _lt(_ROWS)
    with pytest.raises(NotImplementedError, match=f"{cls}.log_abs_det_jacobian"):
        transform.log_abs_det_jacobian(x, transform(x))


def test_transformed_distribution_over_softmax_has_no_density(ref: Any) -> None:
    """The consequence: a density through a non-injective map is undefined, as in the reference."""
    base = D.Independent(D.Normal(_lt([0.0, 0.0, 0.0]), _lt([1.0, 1.0, 1.0])), 1)
    point = _lt([0.2, 0.3, 0.5])
    ours = D.TransformedDistribution(base, [T.SoftmaxTransform()])
    theirs = ref.distributions.TransformedDistribution(
        ref.distributions.Independent(ref.distributions.Normal(ref.zeros(3), ref.ones(3)), 1),
        [ref.distributions.transforms.SoftmaxTransform()],
    )
    with pytest.raises(NotImplementedError):
        ours.log_prob(point)
    with pytest.raises(NotImplementedError):
        theirs.log_prob(_rt(ref, [0.2, 0.3, 0.5]))


def test_transform_to_simplex_still_maps_onto_the_simplex(ref: Any) -> None:
    """Only the determinant went; the map itself is the reference's."""
    from lucid.distributions import constraints as C

    ours = D.transform_to(C.simplex)(_lt(_ROWS))
    theirs = ref.distributions.transform_to(ref.distributions.constraints.simplex)(_rt(ref, _ROWS))
    np.testing.assert_allclose(_arr(ours), _arr(theirs), rtol=3e-4, atol=1e-6)


# ── A number operand follows the input's dtype ───────────────────────────────

_PLACES = [("cpu", np.float32), ("cpu", np.float64), ("metal", np.float32)]
_PLACE_IDS = ["cpu-f32", "cpu-f64", "metal-f32"]

# Exact in neither float32 nor float64 (0.3, 1/3, 1e-3), exactly representable
# (0.5, 2), negative, zero loc, and ints.
_AFFINE_NUMBERS = [(-1.0, 0.3), (0.0, 1.0 / 3.0), (2.5, -0.7), (1e-3, 1e3), (0.1, 0.5), (3, -2)]
_X = [0.4, -2.5, 0.0, 7.25, 1e-3]


def _tol(dtype: Any) -> dict[str, float]:
    return {"rtol": 1e-14, "atol": 0.0} if dtype == np.float64 else {"rtol": 3e-4, "atol": 1e-5}


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("numbers", _AFFINE_NUMBERS, ids=lambda n: f"{n[0]}-{n[1]}")
def test_affine_with_numbers_matches_reference_at_the_inputs_precision(
    numbers: tuple[float, float], place: Any, ref: Any
) -> None:
    device, dtype = place
    loc, scale = numbers
    ours = T.AffineTransform(loc, scale)
    theirs = ref.distributions.transforms.AffineTransform(loc, scale)
    xl, xr = _lt(_X, device, dtype), _rt(ref, _X, dtype)
    y = ours(xl)
    assert y.dtype == xl.dtype and y.device == xl.device
    np.testing.assert_allclose(_arr(y), _arr(theirs(xr)), **_tol(dtype))
    # The inverse takes the reference's own output back to the input.
    ry = theirs(xr)
    np.testing.assert_allclose(
        _arr(ours.inv(_lt(_arr(ry), device, dtype))), _arr(theirs.inv(ry)), **_tol(dtype)
    )
    np.testing.assert_allclose(
        _arr(ours.log_abs_det_jacobian(xl, y)),
        _arr(theirs.log_abs_det_jacobian(xr, ry)),
        **_tol(dtype),
    )
    assert ours.sign == theirs.sign


@pytest.mark.parametrize("numbers", _AFFINE_NUMBERS, ids=lambda n: f"{n[0]}-{n[1]}")
def test_affine_with_numbers_is_the_python_arithmetic_on_float64(
    numbers: tuple[float, float],
) -> None:
    """The reference's number path is plain float64 arithmetic; so is ours, bit for bit."""
    loc, scale = numbers
    x = np.asarray(_X, dtype=np.float64)
    ours = T.AffineTransform(loc, scale)
    got = _arr(ours(lucid.tensor(x, dtype=lucid.float64)))
    np.testing.assert_array_equal(got, loc + scale * x)
    np.testing.assert_allclose(
        _arr(ours.inv(lucid.tensor(got, dtype=lucid.float64))), x, rtol=1e-14, atol=1e-14
    )


def test_affine_zero_scale_log_det_is_minus_infinity() -> None:
    """``math.log(0)`` raises; a held number must not turn ``-inf`` into an error."""
    x = _lt(_VEC)
    out = _arr(T.AffineTransform(0.0, 0.0).log_abs_det_jacobian(x, x))
    assert np.all(np.isneginf(out))


def test_affine_event_dim_still_sums_a_number_scale() -> None:
    x = _lt(_ROWS)
    out = _arr(T.AffineTransform(0.0, 2.0, event_dim=1).log_abs_det_jacobian(x, x))
    np.testing.assert_allclose(out, [3 * np.log(2.0)] * 2, rtol=1e-6)


def test_affine_with_tensor_parameters_is_unchanged(ref: Any) -> None:
    loc, scale = [0.5, -1.0, 2.0], [0.3, -0.7, 1.5]
    ours = T.AffineTransform(_lt(loc, dtype=np.float64), _lt(scale, dtype=np.float64))
    theirs = ref.distributions.transforms.AffineTransform(
        _rt(ref, loc, np.float64), _rt(ref, scale, np.float64)
    )
    x = _lt(_VEC, dtype=np.float64)
    np.testing.assert_allclose(
        _arr(ours(x)), _arr(theirs(_rt(ref, _VEC, np.float64))), rtol=1e-14, atol=0.0
    )


# The reference rounds a ``PowerTransform`` exponent given as a number to its
# default float dtype and keeps it, so its float64 results use that rounded
# exponent; Lucid does the same, and is held to it (not to the exact number).
_EXPONENTS = [0.3, 1.0 / 3.0, -1.5, 2.0, 1.0]


@pytest.mark.parametrize("place", _PLACES, ids=_PLACE_IDS)
@pytest.mark.parametrize("exponent", _EXPONENTS)
def test_power_with_a_number_matches_reference_at_the_inputs_precision(
    exponent: float, place: Any, ref: Any
) -> None:
    device, dtype = place
    ours = T.PowerTransform(exponent)
    theirs = ref.distributions.transforms.PowerTransform(exponent)
    xl, xr = _lt(_POS, device, dtype), _rt(ref, _POS, dtype)
    tol = {"rtol": 1e-12, "atol": 1e-14} if dtype == np.float64 else _tol(dtype)
    y = ours(xl)
    assert y.dtype == xl.dtype
    np.testing.assert_allclose(_arr(y), _arr(theirs(xr)), **tol)
    np.testing.assert_allclose(_arr(ours.inv(y)), _arr(theirs.inv(theirs(xr))), **tol)
    np.testing.assert_allclose(
        _arr(ours.log_abs_det_jacobian(xl, y)),
        _arr(theirs.log_abs_det_jacobian(xr, theirs(xr))),
        **tol,
    )
