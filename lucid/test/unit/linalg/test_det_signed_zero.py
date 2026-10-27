"""The determinant of a singular matrix is a signed zero, not a literal ``+0``.

LU with partial pivoting gives ``det = sign(P) * prod(diag(U))``, and a
singular matrix still factors to the end with an exact zero on that diagonal,
so the product is a signed zero.  The CPU returned a literal ``+0.0`` whenever
the factorisation met a zero pivot, where Metal and the reference return the
signed zero (LCD-301); ``slogdet``'s gradient, ``g / det``, took the wrong
sign of infinity with it.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

#: name -> (matrix, the sign of its zero det)
_SINGULAR: dict[str, tuple[list[list[float]], float]] = {
    "one row swap": ([[1.0, 2.0], [2.0, 4.0]], -1.0),
    "negative pivot": ([[-2.0, 4.0], [1.0, -2.0]], -1.0),
    "swap and negative pivot": ([[-1.0, 2.0], [2.0, -4.0]], -1.0),
    "positive pivots": ([[4.0, 2.0], [2.0, 1.0]], 1.0),
    "zero matrix": ([[0.0, 0.0], [0.0, 0.0]], 1.0),
    "swap in Aᵀ only": ([[2.0, 4.0], [1.0, 2.0]], -1.0),
    "zero column": ([[0.0, 1.0], [0.0, 2.0]], -1.0),
}

_NEGATIVE_ZERO_MATRIX = [[-0.0, 0.0], [0.0, 0.0]]


def _placements() -> list[object]:
    metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")
    return [
        pytest.param("cpu", "float32", id="cpu-float32"),
        pytest.param("cpu", "float64", id="cpu-float64"),
        pytest.param("metal", "float32", id="metal-float32", marks=metal),
    ]


def _tensor(rows: object, device: str, dtype: str) -> lucid.Tensor:
    return lucid.tensor(rows, dtype=getattr(lucid, dtype), device=device)


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("case", list(_SINGULAR))
def test_a_singular_det_is_a_signed_zero(case: str, device: str, dtype: str) -> None:
    rows, sign = _SINGULAR[case]
    A = _tensor(rows, device, dtype)
    d = lucid.linalg.det(A).item()
    s = lucid.linalg.slogdet(A)[0].item()
    assert d == 0.0 and math.copysign(1.0, d) == sign
    assert s == 0.0 and math.copysign(1.0, s) == sign


@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("case", list(_SINGULAR))
def test_the_slogdet_gradient_takes_the_sign_of_the_zero(
    case: str, device: str, dtype: str
) -> None:
    # g / det along the adjugate: [[d, -c], [-b, a]] / (±0).
    rows, sign = _SINGULAR[case]
    A = _tensor(rows, device, dtype).requires_grad_()
    _, logabs = lucid.linalg.slogdet(A)
    logabs.backward()
    assert A.grad is not None
    (a, b), (c, d) = rows
    want = [
        [math.copysign(math.inf, v * sign) if v else math.nan for v in row]
        for row in ([d, -c], [-b, a])
    ]
    got = A.grad.to("cpu").tolist()
    for g_row, w_row in zip(got, want, strict=True):
        for g, w in zip(g_row, w_row, strict=True):
            assert (math.isnan(g) and math.isnan(w)) or g == w, (got, want)


@pytest.mark.parametrize(("device", "dtype"), _placements())
def test_a_batch_mixing_singular_and_regular_matrices(device: str, dtype: str) -> None:
    rows = [_SINGULAR["one row swap"][0], [[3.0, 1.0], [1.0, 2.0]]]
    singular, regular = lucid.linalg.det(_tensor(rows, device, dtype)).to("cpu").tolist()
    assert singular == 0.0 and math.copysign(1.0, singular) == -1.0
    assert math.isclose(regular, 5.0, rel_tol=1e-6)


def _parity_cases() -> list[object]:
    return [
        *(pytest.param(rows, id=name) for name, (rows, _) in _SINGULAR.items()),
        pytest.param(_NEGATIVE_ZERO_MATRIX, id="negative zero matrix"),
    ]


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _placements())
@pytest.mark.parametrize("rows", _parity_cases())
def test_the_zero_has_the_references_sign(
    rows: list[list[float]], device: str, dtype: str, ref: ModuleType
) -> None:
    A = _tensor(rows, device, dtype)
    A_ref = ref.tensor(rows, dtype=getattr(ref, dtype))
    d = lucid.linalg.det(A).item()
    want = float(ref.linalg.det(A_ref))
    assert d == want == 0.0
    assert math.copysign(1.0, d) == math.copysign(1.0, want)
    s = lucid.linalg.slogdet(A)[0].item()
    want_s = float(ref.linalg.slogdet(A_ref).sign)
    assert s == want_s == 0.0
    assert math.copysign(1.0, s) == math.copysign(1.0, want_s)
